from __future__ import annotations

import torch


def select_device(requested: str) -> torch.device:
    """Choose torch device based on request + availability."""
    requested = requested.lower()

    def mps_available() -> bool:
        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if mps_available():
            return torch.device("mps")
        return torch.device("cpu")

    # allow "cuda", "cuda:0", "cpu", "mps"
    dev = torch.device(requested)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"{requested} is requested, but CUDA is not available")
    if dev.type == "mps" and not mps_available():
        raise ValueError(f"{requested} is requested, but MPS is not available")
    return dev
