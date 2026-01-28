# interface definition

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Any

import torch


class ActorCritic(torch.nn.Module, ABC):
    @abstractmethod
    def initial_state(self, batch_size: int, device: torch.device) -> Optional[Any]: ...
    @abstractmethod
    def forward(
        self,
        obs: torch.Tensor,
        # state: Optional[Any],
        # done: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Any]]:
        """Returns (action, logp, value, next_state)."""
        ...

    @abstractmethod
    def evaluate_actions(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        state: Optional[Any],
        done: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (logp, entropy, value)."""
        ...
