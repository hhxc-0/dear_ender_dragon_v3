# factory of models

from __future__ import annotations

from typing_extensions import Self

import torch
from gymnasium import spaces

from .base import ActorCritic
from .mlp_actor_critic import MLPActorCritic
from .cnn_actor_critic import CNNActorCritic
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
    if model_cfg.name == "cnn_actor_critic":
        return CNNActorCritic(
            single_obs_space,
            single_act_space,
            cnn_channels=model_cfg.cnn_channels,
            cnn_kernel_sizes=model_cfg.cnn_kernel_sizes,
            cnn_strides=model_cfg.cnn_strides,
            cnn_paddings=model_cfg.cnn_paddings,
            cnn_activation=model_cfg.cnn_activation,
            mlp_hidden_sizes=model_cfg.mlp_hidden_sizes,
            mlp_activation=model_cfg.mlp_activation,
            shared_backbone=model_cfg.shared_backbone,
            orthogonal_init=model_cfg.orthogonal_init,
        )

    else:
        raise ValueError(f"Unknown model name: {model_cfg.name}")
