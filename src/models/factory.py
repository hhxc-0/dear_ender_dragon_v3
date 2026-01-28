# factory of models

from __future__ import annotations

from typing_extensions import Self

import torch
from gymnasium import spaces

from .base import ActorCritic
from .mlp_actor_critic import MLPActorCritic
from src.types import ModelCfg


def make_model(
    model_cfg: ModelCfg,
    single_obs_space: spaces.Space,
    single_act_space: spaces.Space,
) -> ActorCritic:
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
