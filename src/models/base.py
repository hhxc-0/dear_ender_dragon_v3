# interface definition

from __future__ import annotations

from typing import Protocol, Tuple, Optional, Any
import torch
from omegaconf import DictConfig
from gymnasium import spaces

from .mlp_actor_critic import MLPActorCritic


class ActorCritic(Protocol):
    def initial_state(self, batch_size: int, device: torch.device) -> Optional[Any]: ...

    def act(
        self,
        obs: torch.Tensor,
        state: Optional[Any],
        done: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Any]]:
        """Returns (action, logp, value, next_state)."""
        ...

    def evaluate_actions(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (logp, entropy, value)."""
        ...


def make_model(
    model_cfg: DictConfig,
    single_obs_space: spaces.Space,
    single_act_space: spaces.Space,
) -> torch.nn.Module:
    if model_cfg.name == "mlp_actor_critic":
        return MLPActorCritic(
            single_obs_space,
            single_act_space,
            hidden_sizes=model_cfg.hidden_sizes,
            activation=model_cfg.activation,
            shared_backbone=model_cfg.shared_backbone,
            orthogonal_init=model_cfg.orthogonal_init,
        )

    else:
        raise ValueError(f"Unknown model name: {model_cfg.name}")
