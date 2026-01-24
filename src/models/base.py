# interface definition

from typing import Protocol, Tuple, Optional, Any
import torch

class PolicyModel(Protocol):
    def initial_state(self, batch_size: int, device: torch.device) -> Optional[Any]:
        ...

    def forward(
        self,
        obs: torch.Tensor,
        state: Optional[Any],
        done: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.distributions.Distribution, torch.Tensor, Optional[Any]]:
        """Returns (action_dist, value, next_state)."""
        ...
