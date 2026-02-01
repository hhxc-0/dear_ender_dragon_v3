# Small dataclasses for Batch/Rollout + Metrics dict typing (optional but helpful)

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Literal, NamedTuple
import torch
from torch import Tensor


@dataclass
class ModelCfg:
    name: Literal["mlp_actor_critic"]
    hidden_sizes: Sequence[int]
    activation: str
    shared_backbone: bool
    orthogonal_init: bool


@dataclass
class PPOCfg:
    clip_coef: float
    ent_coef: float
    vf_coef: float
    max_grad_norm: float


class MiniBatch(NamedTuple):
    batch_size: int
    obs: Tensor
    actions: Tensor
    logp_old: Tensor
    values_old: Tensor
    returns: Tensor
    advantages_norm: Tensor
