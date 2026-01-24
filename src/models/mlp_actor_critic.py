# MLP actor-critic for CartPole/LunarLander (Categorical actions)

from typing import Tuple, Optional, Any, Callable
import torch
from torch import nn


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


class MLPActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        *,
        hidden_sizes: tuple[int, ...] = (64, 64),
        activation: str = "tanh",  # "tanh" (SB3-style) or "relu"
        shared_backbone: bool = True,  # shared trunk vs separate pi/v networks
        orthogonal_init: bool = True,  # common in PPO
        log_std_init: float = -0.5,  # only used if you later add continuous actions
    ):
        super().__init__()
        self.shared_backbone = shared_backbone
        if activation == "tanh":
            make_act = lambda: nn.Tanh()
        elif activation == "relu":
            make_act = lambda: nn.ReLU(inplace=True)
        else:
            raise ValueError(f"Unknown activation: {activation}")

        if self.shared_backbone:
            self.trunk_mlp = make_mlp(
                (obs_dim, *hidden_sizes), make_act, activate_last=True
            )
            self.pi_head = nn.Linear(hidden_sizes[-1], act_dim)
            self.v_head = nn.Linear(hidden_sizes[-1], 1)
        else:
            self.pi_mlp = make_mlp((obs_dim, *hidden_sizes), make_act)
            self.pi_head = nn.Linear(hidden_sizes[-1], act_dim)
            self.v_mlp = make_mlp((obs_dim, *hidden_sizes), make_act)
            self.v_head = nn.Linear(hidden_sizes[-1], 1)

        # TODO: initialization

    def initial_state(self, batch_size: int, device: torch.device) -> Optional[Any]:
        return None

    def forward(
        self,
        obs: torch.Tensor,
        state: Optional[Any],
        done: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.distributions.Distribution, torch.Tensor, Optional[Any]]:
        """Returns (action_dist, value, next_state)."""
        ...
