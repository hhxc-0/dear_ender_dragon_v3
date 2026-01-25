# MLP actor-critic for CartPole/LunarLander (Categorical actions)

from typing import Tuple, Optional, Any, Callable
import math
import torch
from torch import nn
from torch.distributions import Categorical


def make_mlp(
    sizes: tuple[int, ...],
    act_maker: Callable[[], nn.Module],
    *,
    activate_last: bool = False,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        is_last = i == len(sizes) - 2
        if (not is_last) or activate_last:
            layers.append(act_maker())
    return nn.Sequential(*layers)


def init_linear_orthogonal(layer: nn.Linear, gain: float) -> None:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)


def init_mlp_orthogonal(module: nn.Module, hidden_gain: float) -> None:
    # Initialize all Linear layers inside module
    for m in module.modules():
        if isinstance(m, nn.Linear):
            init_linear_orthogonal(m, gain=hidden_gain)


def gain_for_activation(activation: str) -> float:
    # Reasonable defaults for PPO
    if activation == "tanh":
        return math.sqrt(2.0)
        # return nn.init.calculate_gain("tanh")  # for later use
    if activation == "relu":
        return math.sqrt(2.0)
        # return nn.init.calculate_gain("relu")  # for later use
    raise ValueError(f"Unknown activation: {activation}")


def activation_factory(activation: str) -> Callable[[], nn.Module]:
    if activation == "tanh":
        return lambda: nn.Tanh()
    if activation == "relu":
        return lambda: nn.ReLU()
    raise ValueError(f"Unknown activation: {activation}")


class MLPActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        *,
        debug: bool = False,
        hidden_sizes: tuple[int, ...] = (64, 64),
        activation: str = "tanh",  # "tanh" (SB3-style) or "relu"
        shared_backbone: bool = True,  # shared trunk vs separate pi/v networks
        orthogonal_init: bool = True,  # common in PPO
        # log_std_init: float = -0.5,  # only used if you later add continuous actions
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.debug = debug
        self.shared_backbone = shared_backbone

        make_act = activation_factory(activation)
        assert len(hidden_sizes) > 0

        # definition
        if shared_backbone:
            self.trunk_mlp = make_mlp(
                (obs_dim, *hidden_sizes), make_act, activate_last=True
            )

        else:
            self.pi_mlp = make_mlp(
                (obs_dim, *hidden_sizes), make_act, activate_last=True
            )
            self.v_mlp = make_mlp(
                (obs_dim, *hidden_sizes), make_act, activate_last=True
            )
        self.pi_head = nn.Linear(hidden_sizes[-1], act_dim)
        self.v_head = nn.Linear(hidden_sizes[-1], 1)

        # initialization
        if orthogonal_init:
            hidden_gain = gain_for_activation(activation)
            if shared_backbone:
                init_mlp_orthogonal(self.trunk_mlp, hidden_gain)
            else:
                init_mlp_orthogonal(self.pi_mlp, hidden_gain)
                init_mlp_orthogonal(self.v_mlp, hidden_gain)
            init_linear_orthogonal(self.pi_head, 0.01)
            init_linear_orthogonal(self.v_head, 1.0)

    def initial_state(self, batch_size: int, device: torch.device) -> Optional[Any]:
        return None

    def forward(
        self,
        obs: torch.Tensor,  # [batch, obs_dim]
        state: Optional[Any],  # unused
        done: Optional[torch.Tensor] = None,  # unused
    ) -> Tuple[torch.distributions.Distribution, torch.Tensor, Optional[Any]]:
        """Returns (action_dist, value, next_state)."""
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

        # post-processing
        action_dist = Categorical(logits=logits)
        value = value.squeeze(-1)  # [batch,1] -> [batch]
        if self.debug:
            assert torch.isfinite(logits).all()
            assert torch.isfinite(value).all()
        return action_dist, value, None
