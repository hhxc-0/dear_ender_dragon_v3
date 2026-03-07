# MLP actor-critic for CartPole/LunarLander (Categorical actions)

from __future__ import annotations

from typing import Tuple, Optional, Any, Sequence
import torch
from torch import nn
from torch.distributions import Categorical
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .base import ActorCritic
from .nn_utils import (
    build_mlp,
    gain_for_activation,
    init_linear_orthogonal,
    init_module_orthogonal,
)


class MLPActorCritic(ActorCritic):
    def __init__(
        self,
        single_obs_space: spaces.Space,
        single_act_space: spaces.Space,
        *,
        debug: bool = False,
        hidden_sizes: Sequence[int] = (64, 64),
        activation: str = "tanh",  # "tanh" (SB3-style) or "relu"
        shared_backbone: bool = True,  # shared trunk vs separate pi/v networks
        orthogonal_init: bool = True,  # common in PPO
        # log_std_init: float = -0.5,  # only used if you later add continuous actions
    ) -> None:
        super().__init__()
        assert isinstance(single_obs_space, spaces.Box)
        assert isinstance(single_act_space, spaces.Discrete)
        assert len(single_obs_space.shape) == 1
        assert isinstance(
            single_act_space.n, (int, np.integer)
        ), f"Expect int for number of actions, got {type(single_act_space.n)}"
        self.obs_dim = single_obs_space.shape[0]
        self.act_dim = int(single_act_space.n)
        self.debug = debug
        self.shared_backbone = shared_backbone
        assert len(hidden_sizes) > 0

        # definition
        if shared_backbone:
            self.trunk_mlp = build_mlp(
                [self.obs_dim, *hidden_sizes],
                activation=activation,
                activate_last=True,
            )

        else:
            self.pi_mlp = build_mlp(
                [self.obs_dim, *hidden_sizes],
                activation=activation,
                activate_last=True,
            )
            self.v_mlp = build_mlp(
                [self.obs_dim, *hidden_sizes],
                activation=activation,
                activate_last=True,
            )
        self.pi_head = nn.Linear(hidden_sizes[-1], self.act_dim)
        self.v_head = nn.Linear(hidden_sizes[-1], 1)

        # initialization
        if orthogonal_init:
            hidden_gain = gain_for_activation(activation)
            if shared_backbone:
                init_module_orthogonal(self.trunk_mlp, hidden_gain)
            else:
                init_module_orthogonal(self.pi_mlp, hidden_gain)
                init_module_orthogonal(self.v_mlp, hidden_gain)
            init_linear_orthogonal(self.pi_head, 0.01)
            init_linear_orthogonal(self.v_head, 1.0)

    def initial_state(self, batch_size: int, device: torch.device) -> Optional[Any]:
        return None

    def get_action_and_value(
        self,
        obs: torch.Tensor,  # [batch, obs_dim]
        # state: Optional[Any],  # unused
        # done: Optional[torch.Tensor] = None,  # unused
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Any]]:
        """Returns (action, logp, value, next_state)."""
        # arguments checking and pre-processing
        if self.debug:
            assert (
                obs.ndim == 2
            ), f"Expected obs [B, obs_dim], got shape {tuple(obs.shape)}"
            assert (
                obs.shape[1] == self.obs_dim
            ), f"Expected obs_dim={self.obs_dim}, got {obs.shape[1]}"
        obs = obs.to(torch.float32)

        # inference
        if self.shared_backbone:
            hidden = self.trunk_mlp(obs)
            logits = self.pi_head(hidden)
            value = self.v_head(hidden)
        else:
            pi_hidden = self.pi_mlp(obs)
            v_hidden = self.v_mlp(obs)
            logits = self.pi_head(pi_hidden)
            value = self.v_head(v_hidden)
        if self.debug:
            assert torch.isfinite(logits).all()
            assert torch.isfinite(value).all()

        # post-processing
        action_dist = Categorical(logits=logits)
        action = action_dist.sample()
        logp = action_dist.log_prob(action)
        value = value.squeeze(-1)  # [batch,1] -> [batch]
        return action, logp, value, None

    def get_value(
        self,
        obs: torch.Tensor,  # [batch, obs_dim]
        # state: Optional[Any],  # unused
        # done: Optional[torch.Tensor] = None,  # unused
    ) -> torch.Tensor:
        """Returns value."""
        # arguments checking and pre-processing
        if self.debug:
            assert (
                obs.ndim == 2
            ), f"Expected obs [B, obs_dim], got shape {tuple(obs.shape)}"
            assert (
                obs.shape[1] == self.obs_dim
            ), f"Expected obs_dim={self.obs_dim}, got {obs.shape[1]}"
        obs = obs.to(torch.float32)

        # inference
        if self.shared_backbone:
            hidden = self.trunk_mlp(obs)
            value = self.v_head(hidden)
        else:
            v_hidden = self.v_mlp(obs)
            value = self.v_head(v_hidden)
        if self.debug:
            assert torch.isfinite(value).all()

        # post-processing
        value = value.squeeze(-1)  # [batch,1] -> [batch]
        return value

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        # state: Optional[Any],  # unused
        # done: Optional[torch.Tensor],  # unused
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (logp, entropy, value)."""
        # arguments checking and pre-processing
        if self.debug:
            assert (
                obs.ndim == 2
            ), f"Expected obs [B, obs_dim], got shape {tuple(obs.shape)}"
            assert (
                obs.shape[1] == self.obs_dim
            ), f"Expected obs_dim={self.obs_dim}, got {obs.shape[1]}"
        obs = obs.to(torch.float32)

        # inference
        if self.shared_backbone:
            hidden = self.trunk_mlp(obs)
            logits = self.pi_head(hidden)
            value = self.v_head(hidden)
        else:
            pi_hidden = self.pi_mlp(obs)
            v_hidden = self.v_mlp(obs)
            logits = self.pi_head(pi_hidden)
            value = self.v_head(v_hidden)
        if self.debug:
            assert torch.isfinite(logits).all()
            assert torch.isfinite(value).all()

        # post-processing
        action_dist = Categorical(logits=logits)
        logp = action_dist.log_prob(action)
        entropy = action_dist.entropy()
        value = value.squeeze(-1)  # [batch,1] -> [batch]
        return logp, entropy, value
