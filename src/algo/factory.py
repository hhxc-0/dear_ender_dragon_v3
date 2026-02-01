# factory of learners

from __future__ import annotations

from omegaconf import DictConfig
import torch
from src.models.base import ActorCritic
from .base import Learner
from .ppo import PPOLearner


def make_learner(cfg: DictConfig, model: ActorCritic, optim: torch.optim.Optimizer) -> Learner:
    if cfg.ppo is not None:
        return PPOLearner(model=model, optim=optim, config=cfg.ppo)
    raise ValueError("Learner not found in config.")