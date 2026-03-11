# CNN actor-critic for image observation (Categorical actions)

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
    build_cnn,
    build_mlp,
    gain_for_activation,
    init_linear_orthogonal,
    init_module_orthogonal,
)


def get_flattened_dim(input_shape: Sequence[int], cnn: torch.nn.Module) -> int:
    dummy = torch.zeros(1, *input_shape)
    output = cnn(dummy)
    assert isinstance(output, torch.Tensor)
    return output.flatten(1).shape[1]


class CNNActorCritic(ActorCritic):
    def __init__(
        self,
        single_obs_space: spaces.Space,
        single_act_space: spaces.Space,
        *,
        debug: bool = False,
        cnn_channels: Sequence[int],
        cnn_kernel_sizes: Sequence[int],
        cnn_strides: Sequence[int],
        cnn_paddings: Sequence[int],
        cnn_activation: str = "relu",  # "tanh" or "relu"
        mlp_hidden_sizes: Sequence[int] = (256,),
        mlp_activation: str = "tanh",  # "tanh" or "relu"
        shared_backbone: bool = True,  # shared trunk vs separate pi/v networks
        orthogonal_init: bool = True,  # common in PPO
        # log_std_init: float = -0.5,  # only used if you later add continuous actions
    ) -> None:
        super().__init__()
        assert isinstance(single_obs_space, spaces.Box)
        assert isinstance(single_act_space, spaces.Discrete)
        assert (
            len(single_obs_space.shape) == 3
        ), f"Expected image obs with shape (C,H,W), got {single_obs_space.shape}"
        assert isinstance(
            single_act_space.n, (int, np.integer)
        ), f"Expect int for number of actions, got {type(single_act_space.n)}"
        self.obs_shape = single_obs_space.shape
        self.act_dim = int(single_act_space.n)
        self.debug = debug
        self.shared_backbone = shared_backbone
        in_channels = self.obs_shape[0]

        # definition
        if shared_backbone:
            self.trunk_cnn = build_cnn(
                channels=[in_channels, *cnn_channels],
                kernel_sizes=cnn_kernel_sizes,
                strides=cnn_strides,
                paddings=cnn_paddings,
                activation=cnn_activation,
                activate_last=True,
            )
            flat_dim = get_flattened_dim(
                input_shape=single_obs_space.shape, cnn=self.trunk_cnn
            )
            self.trunk_mlp = build_mlp(
                [flat_dim, *mlp_hidden_sizes],
                activation=mlp_activation,
                activate_last=True,
            )

        else:
            self.pi_cnn = build_cnn(
                channels=[in_channels, *cnn_channels],
                kernel_sizes=cnn_kernel_sizes,
                strides=cnn_strides,
                paddings=cnn_paddings,
                activation=cnn_activation,
                activate_last=True,
            )
            pi_flat_dim = get_flattened_dim(
                input_shape=single_obs_space.shape, cnn=self.pi_cnn
            )
            self.pi_mlp = build_mlp(
                [pi_flat_dim, *mlp_hidden_sizes],
                activation=mlp_activation,
                activate_last=True,
            )

            self.v_cnn = build_cnn(
                channels=[in_channels, *cnn_channels],
                kernel_sizes=cnn_kernel_sizes,
                strides=cnn_strides,
                paddings=cnn_paddings,
                activation=cnn_activation,
                activate_last=True,
            )
            v_flat_dim = get_flattened_dim(
                input_shape=single_obs_space.shape, cnn=self.v_cnn
            )
            self.v_mlp = build_mlp(
                [v_flat_dim, *mlp_hidden_sizes],
                activation=mlp_activation,
                activate_last=True,
            )
        self.pi_head = nn.Linear(mlp_hidden_sizes[-1], self.act_dim)
        self.v_head = nn.Linear(mlp_hidden_sizes[-1], 1)

        # initialization
        if orthogonal_init:
            cnn_gain = gain_for_activation(cnn_activation)
            mlp_gain = gain_for_activation(mlp_activation)
            if shared_backbone:
                init_module_orthogonal(self.trunk_cnn, cnn_gain)
                init_module_orthogonal(self.trunk_mlp, mlp_gain)
            else:
                init_module_orthogonal(self.pi_cnn, cnn_gain)
                init_module_orthogonal(self.pi_mlp, mlp_gain)
                init_module_orthogonal(self.v_cnn, cnn_gain)
                init_module_orthogonal(self.v_mlp, mlp_gain)
            init_linear_orthogonal(self.pi_head, 0.01)
            init_linear_orthogonal(self.v_head, 1.0)

    def initial_state(self, batch_size: int, device: torch.device) -> Optional[Any]:
        return None

    def _check_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Validate obs shape (debug only) and cast to float32."""
        if self.debug:
            assert obs.ndim == 1 + len(
                self.obs_shape
            ), f"Expected obs [B, *obs_shape], got shape {tuple(obs.shape)}"
            assert (
                obs.shape[1:] == self.obs_shape
            ), f"Expected obs.shape={[1, *self.obs_shape]}, got {obs.shape}"
        return obs.to(torch.float32)

    def _forward_pi(self, obs: torch.Tensor) -> torch.Tensor:
        """Policy-only forward pass returning logits."""
        if self.shared_backbone:
            hidden = self.trunk_cnn(obs)
            hidden = hidden.flatten(1)
            hidden = self.trunk_mlp(hidden)
            logits = self.pi_head(hidden)
        else:
            pi_hidden = self.pi_cnn(obs)
            pi_hidden = pi_hidden.flatten(1)
            pi_hidden = self.pi_mlp(pi_hidden)
            logits = self.pi_head(pi_hidden)
        if self.debug:
            assert torch.isfinite(logits).all()
        return logits

    def _forward_v(self, obs: torch.Tensor) -> torch.Tensor:
        """Value-only forward pass returning value."""
        if self.shared_backbone:
            hidden = self.trunk_cnn(obs)
            hidden = hidden.flatten(1)
            hidden = self.trunk_mlp(hidden)
            value = self.v_head(hidden)
        else:
            v_hidden = self.v_cnn(obs)
            v_hidden = v_hidden.flatten(1)
            v_hidden = self.v_mlp(v_hidden)
            value = self.v_head(v_hidden)
        if self.debug:
            assert torch.isfinite(value).all()
        return value

    def get_action_and_value(
        self,
        obs: torch.Tensor,  # [batch, *obs_shape]
        # state: Optional[Any],  # unused
        # done: Optional[torch.Tensor] = None,  # unused
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Any]]:
        """Returns (action, logp, value, next_state)."""
        obs = self._check_obs(obs)
        logits = self._forward_pi(obs)
        value = self._forward_v(obs)
        action_dist = Categorical(logits=logits)
        action = action_dist.sample()
        logp = action_dist.log_prob(action)
        value = value.squeeze(-1)  # [batch,1] -> [batch]
        return action, logp, value, None

    def get_value(
        self,
        obs: torch.Tensor,  # [batch, *obs_shape]
        # state: Optional[Any],  # unused
        # done: Optional[torch.Tensor] = None,  # unused
    ) -> torch.Tensor:
        """Returns value."""
        obs = self._check_obs(obs)
        value = self._forward_v(obs)
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
        obs = self._check_obs(obs)
        logits = self._forward_pi(obs)
        value = self._forward_v(obs)
        action_dist = Categorical(logits=logits)
        logp = action_dist.log_prob(action)
        entropy = action_dist.entropy()
        value = value.squeeze(-1)  # [batch,1] -> [batch]
        return logp, entropy, value
