from __future__ import annotations

from typing import Any, Optional, Union

import torch


def to_torch(
    x: Any, device: Union[str, torch.device], dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    # torch.as_tensor avoids an extra copy on CPU when possible;
    # specifying device will move/copy to GPU if needed.
    t = torch.as_tensor(x, device=device)
    return t if dtype is None else t.to(dtype)
