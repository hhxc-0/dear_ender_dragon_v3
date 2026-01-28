# Small dataclasses for Batch/Rollout + Metrics dict typing (optional but helpful)

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Literal

@dataclass
class ModelCfg:
    name: Literal["mlp_actor_critic"]
    hidden_sizes: Sequence[int]
    activation: str
    shared_backbone: bool
    orthogonal_init: bool
