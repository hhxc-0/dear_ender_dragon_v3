# Shared fixtures for all unit tests

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import gymnasium as gym
from gymnasium import spaces

from src.models.mlp_actor_critic import MLPActorCritic
from src.buffers.rollout_buffer import RolloutBuffer
from src.types import MiniBatch


# ---------------------------------------------------------------------------
# Spaces
# ---------------------------------------------------------------------------

@pytest.fixture
def obs_space() -> spaces.Box:
    """CartPole-like 4-dim continuous observation space."""
    return spaces.Box(low=-4.0, high=4.0, shape=(4,), dtype=np.float32)


@pytest.fixture
def act_space() -> spaces.Discrete:
    """CartPole-like 2-action discrete action space."""
    return spaces.Discrete(2)


# ---------------------------------------------------------------------------
# Model + optimiser
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_model(obs_space, act_space) -> MLPActorCritic:
    """Small shared-backbone MLP actor-critic on CPU."""
    return MLPActorCritic(
        single_obs_space=obs_space,
        single_act_space=act_space,
        debug=False,
        hidden_sizes=(8,),
        activation="tanh",
        shared_backbone=True,
        orthogonal_init=False,
    )


@pytest.fixture
def tiny_model_ortho(obs_space, act_space) -> MLPActorCritic:
    """Small shared-backbone MLP with orthogonal init, for init tests."""
    return MLPActorCritic(
        single_obs_space=obs_space,
        single_act_space=act_space,
        debug=False,
        hidden_sizes=(16,),
        activation="tanh",
        shared_backbone=True,
        orthogonal_init=True,
    )


@pytest.fixture
def tiny_optim(tiny_model) -> torch.optim.Adam:
    return torch.optim.Adam(tiny_model.parameters(), lr=1e-3)


# ---------------------------------------------------------------------------
# PPO config stub
# ---------------------------------------------------------------------------

@pytest.fixture
def ppo_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        clip_coef=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
    )


# ---------------------------------------------------------------------------
# Buffer helpers
# ---------------------------------------------------------------------------

T, B, OBS_DIM = 4, 2, 4


def _make_filled_buffer(
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    normalize_advantages: bool = False,
    eps: float = 1e-8,
) -> RolloutBuffer:
    """Return a fully-filled, finalized, and computed RolloutBuffer."""
    buf = RolloutBuffer(
        T=T,
        B=B,
        obs_shape=(OBS_DIM,),
        obs_dtype=torch.float32,
        reward_dtype=torch.float32,
        action_shape=(),
        device="cpu",
    )
    torch.manual_seed(0)
    for t in range(T):
        buf.add(
            obs=torch.randn(B, OBS_DIM),
            action=torch.zeros(B, dtype=torch.int64),
            logp=torch.full((B,), -0.693),
            value=torch.ones(B) * 0.5,
            next_value=torch.zeros(B),  # placeholder; overwritten by finalize
            reward=torch.ones(B),
            terminated=torch.zeros(B, dtype=torch.bool),
            truncated=torch.zeros(B, dtype=torch.bool),
            done=torch.zeros(B, dtype=torch.bool),
            timeout=torch.zeros(B, dtype=torch.bool),
            episode_start=torch.zeros(B, dtype=torch.bool),
        )
    last_value = torch.ones(B) * 0.5
    buf.finalize(last_value)
    buf.compute_returns_and_advantages(
        gamma=gamma,
        gae_lambda=gae_lambda,
        normalize_advantages=normalize_advantages,
        eps=eps,
    )
    return buf


@pytest.fixture
def filled_buffer() -> RolloutBuffer:
    return _make_filled_buffer()


# ---------------------------------------------------------------------------
# MiniBatch helper
# ---------------------------------------------------------------------------

def make_minibatch(
    B: int = 8,
    obs_dim: int = 4,
    logp_offset: float = 0.0,
    adv: float = 1.0,
    value_offset: float = 0.0,
) -> MiniBatch:
    """
    Construct a synthetic MiniBatch for PPO update tests.

    logp_offset: logp = logp_old + logp_offset  (ratio = exp(logp_offset))
    adv:         constant advantages_norm value
    value_offset: value = values_old + value_offset
    """
    torch.manual_seed(42)
    obs = torch.randn(B, obs_dim)
    actions = torch.zeros(B, dtype=torch.int64)
    logp_old = torch.full((B,), -0.693)
    values_old = torch.ones(B) * 1.0
    returns = torch.ones(B) * 1.5
    advantages_norm = torch.full((B,), adv)
    return MiniBatch(
        batch_size=B,
        obs=obs,
        actions=actions,
        logp_old=logp_old,
        values_old=values_old,
        returns=returns,
        advantages_norm=advantages_norm,
    )
