from __future__ import annotations

from typing import Callable, Sequence

import torch.nn as nn


def activation_factory(activation: str) -> Callable[[], nn.Module]:
    """Return a zero-arg constructor for the requested activation."""
    if activation == "tanh":
        return lambda: nn.Tanh()
    if activation == "relu":
        return lambda: nn.ReLU()
    raise ValueError(f"Unknown activation: {activation}")


def gain_for_activation(activation: str) -> float:
    """Orthogonal init gain defaults aligned with common PPO practice."""
    # NOTE: Stable-Baselines3 historically uses sqrt(2) for both tanh and relu.
    # Here we prefer PyTorch's canonical gains for the named non-linearities.
    if activation in ("tanh", "relu"):
        return float(nn.init.calculate_gain(activation))
    raise ValueError(f"Unknown activation: {activation}")


def init_linear_orthogonal(layer: nn.Linear, gain: float) -> None:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)


def init_conv2d_orthogonal(layer: nn.Conv2d, gain: float) -> None:
    nn.init.orthogonal_(layer.weight, gain=gain)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, 0.0)


def init_module_orthogonal(module: nn.Module, hidden_gain: float) -> None:
    """Initialize all Linear/Conv2d layers inside module with orthogonal weights."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            init_linear_orthogonal(m, gain=hidden_gain)
        elif isinstance(m, nn.Conv2d):
            init_conv2d_orthogonal(m, gain=hidden_gain)


def build_mlp(
    layer_sizes: Sequence[int],
    *,
    activation: str,
    activate_last: bool = False,
) -> nn.Sequential:
    """Build a simple MLP: Linear layers with an activation after each layer."""
    if len(layer_sizes) < 2:
        raise ValueError(
            f"layer_sizes must have at least 2 entries [in_dim, out_dim], got {layer_sizes}"
        )

    act_maker = activation_factory(activation)

    layers: list[nn.Module] = []
    for i in range(len(layer_sizes) - 1):
        layers.append(nn.Linear(int(layer_sizes[i]), int(layer_sizes[i + 1])))
        is_last = i == len(layer_sizes) - 2
        if (not is_last) or activate_last:
            layers.append(act_maker())
    return nn.Sequential(*layers)


def build_cnn(
    channels: Sequence[int],
    kernel_sizes: Sequence[int],
    strides: Sequence[int],
    paddings: Sequence[int],
    *,
    activation: str,
    activate_last: bool = True,
) -> nn.Sequential:
    """Build a simple CNN: Conv2d layers with an activation after each layer."""
    assert len(channels) >= 2
    assert len(kernel_sizes) >= 1
    assert len(strides) >= 1
    assert len(paddings) >= 1

    act_maker = activation_factory(activation)

    layers: list[nn.Module] = []
    for i in range(len(channels) - 1):
        layers.append(
            nn.Conv2d(
                int(channels[i]),
                int(channels[i + 1]),
                int(kernel_sizes[i]),
                int(strides[i]),
                int(paddings[i]),
            )
        )
        is_last = i == len(channels) - 2
        if (not is_last) or activate_last:
            layers.append(act_maker())
    return nn.Sequential(*layers)
