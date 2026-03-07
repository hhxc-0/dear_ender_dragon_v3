# Unit tests for src/models/mlp_actor_critic.py

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from gymnasium import spaces

from src.models.mlp_actor_critic import MLPActorCritic
from src.models.nn_utils import build_mlp, gain_for_activation, activation_factory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

OBS_DIM = 4
ACT_N = 2
B = 6  # batch size


def _obs_space(dim=OBS_DIM):
    return spaces.Box(low=-1.0, high=1.0, shape=(dim,), dtype=np.float32)


def _act_space(n=ACT_N):
    return spaces.Discrete(n)


def _model(shared=True, ortho=False, debug=False, hidden=(8,), activation="tanh"):
    return MLPActorCritic(
        single_obs_space=_obs_space(),
        single_act_space=_act_space(),
        debug=debug,
        hidden_sizes=hidden,
        activation=activation,
        shared_backbone=shared,
        orthogonal_init=ortho,
    )


def _rand_obs(batch=B):
    torch.manual_seed(0)
    return torch.randn(batch, OBS_DIM)


# ---------------------------------------------------------------------------
# Construction guards
# ---------------------------------------------------------------------------


class TestConstructionGuards:
    def test_non_box_obs_raises(self):
        with pytest.raises(AssertionError):
            MLPActorCritic(
                single_obs_space=spaces.Discrete(4),
                single_act_space=_act_space(),
            )

    def test_non_discrete_act_raises(self):
        with pytest.raises(AssertionError):
            MLPActorCritic(
                single_obs_space=_obs_space(),
                single_act_space=spaces.Box(low=-1.0, high=1.0, shape=(2,)),
            )

    def test_2d_obs_raises(self):
        with pytest.raises(AssertionError):
            MLPActorCritic(
                single_obs_space=spaces.Box(low=-1.0, high=1.0, shape=(4, 4), dtype=np.float32),
                single_act_space=_act_space(),
            )

    def test_unknown_activation_raises(self):
        with pytest.raises(ValueError):
            _model(activation="sigmoid")


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------


class TestArchitecture:
    def test_shared_backbone_has_trunk_mlp(self):
        m = _model(shared=True)
        assert hasattr(m, "trunk_mlp")
        assert not hasattr(m, "pi_mlp")
        assert not hasattr(m, "v_mlp")

    def test_separate_backbone_has_pi_v_mlp(self):
        m = _model(shared=False)
        assert hasattr(m, "pi_mlp")
        assert hasattr(m, "v_mlp")
        assert not hasattr(m, "trunk_mlp")

    def test_has_pi_head_and_v_head(self):
        for shared in (True, False):
            m = _model(shared=shared)
            assert hasattr(m, "pi_head")
            assert hasattr(m, "v_head")

    def test_pi_head_output_dim(self):
        m = _model()
        assert m.pi_head.out_features == ACT_N

    def test_v_head_output_dim(self):
        m = _model()
        assert m.v_head.out_features == 1


# ---------------------------------------------------------------------------
# Orthogonal initialisation
# ---------------------------------------------------------------------------


class TestOrthogonalInit:
    def test_hidden_weights_approximately_orthonormal(self, tiny_model_ortho):
        """Columns of each hidden weight matrix should be approximately orthonormal."""
        import torch.nn as nn
        for module in tiny_model_ortho.modules():
            if isinstance(module, nn.Linear) and module is not tiny_model_ortho.pi_head and module is not tiny_model_ortho.v_head:
                W = module.weight  # [out, in]
                # W @ W^T should be close to I (for square or tall matrices)
                if W.shape[0] <= W.shape[1]:
                    gram = W @ W.T
                    eye = torch.eye(gram.shape[0])
                    assert torch.allclose(gram, eye, atol=1e-5), f"Non-orthonormal weight in {module}"

    def test_pi_head_weight_norm_small(self, tiny_model_ortho):
        """pi_head is initialised with gain=0.01 → weight norms should be tiny."""
        pi_norm = tiny_model_ortho.pi_head.weight.norm().item()
        assert pi_norm < 0.5, f"pi_head weight norm too large: {pi_norm}"

    def test_v_head_weight_norm_order_one(self, tiny_model_ortho):
        """v_head is initialised with gain=1.0 → weight norm should be ~1."""
        v_norm = tiny_model_ortho.v_head.weight.norm().item()
        assert 0.1 < v_norm < 10.0, f"v_head weight norm unexpected: {v_norm}"

    def test_pi_head_norm_much_smaller_than_hidden(self, tiny_model_ortho):
        """pi_head (gain=0.01) should have much smaller norm than hidden layers (gain=sqrt(2))."""
        import torch.nn as nn
        hidden_norms = [
            m.weight.norm().item()
            for m in tiny_model_ortho.modules()
            if isinstance(m, nn.Linear) and m is not tiny_model_ortho.pi_head and m is not tiny_model_ortho.v_head
        ]
        pi_norm = tiny_model_ortho.pi_head.weight.norm().item()
        assert all(pi_norm < hn for hn in hidden_norms)

    def test_biases_initialised_to_zero(self, tiny_model_ortho):
        import torch.nn as nn
        for m in tiny_model_ortho.modules():
            if isinstance(m, nn.Linear):
                assert torch.allclose(m.bias, torch.zeros_like(m.bias)), "Bias not zero"


# ---------------------------------------------------------------------------
# get_action_and_value
# ---------------------------------------------------------------------------


class TestGetActionAndValue:
    def test_action_shape(self):
        m = _model()
        obs = _rand_obs()
        action, logp, value, state = m.get_action_and_value(obs)
        assert action.shape == (B,)

    def test_logp_shape(self):
        m = _model()
        obs = _rand_obs()
        _, logp, _, _ = m.get_action_and_value(obs)
        assert logp.shape == (B,)

    def test_value_shape(self):
        m = _model()
        obs = _rand_obs()
        _, _, value, _ = m.get_action_and_value(obs)
        assert value.shape == (B,)

    def test_state_is_none(self):
        m = _model()
        obs = _rand_obs()
        _, _, _, state = m.get_action_and_value(obs)
        assert state is None

    def test_logp_is_log_probability(self):
        """log-probabilities must be ≤ 0."""
        m = _model()
        obs = _rand_obs()
        _, logp, _, _ = m.get_action_and_value(obs)
        assert (logp <= 0).all()

    def test_action_in_valid_range(self):
        m = _model()
        obs = _rand_obs()
        action, _, _, _ = m.get_action_and_value(obs)
        assert (action >= 0).all() and (action < ACT_N).all()

    def test_obs_cast_to_float32(self):
        """Integer-dtype obs should be accepted and produce float32 outputs."""
        m = _model()
        obs_int = torch.randint(0, 4, (B, OBS_DIM))
        action, logp, value, _ = m.get_action_and_value(obs_int)
        assert logp.dtype == torch.float32
        assert value.dtype == torch.float32

    def test_separate_backbone_same_interface(self):
        m = _model(shared=False)
        obs = _rand_obs()
        action, logp, value, state = m.get_action_and_value(obs)
        assert action.shape == (B,)
        assert logp.shape == (B,)
        assert value.shape == (B,)
        assert state is None


# ---------------------------------------------------------------------------
# get_value
# ---------------------------------------------------------------------------


class TestGetValue:
    def test_value_shape(self):
        m = _model()
        obs = _rand_obs()
        value = m.get_value(obs)
        assert value.shape == (B,)

    def test_value_dtype(self):
        m = _model()
        obs = _rand_obs()
        value = m.get_value(obs)
        assert value.dtype == torch.float32

    def test_value_consistent_with_get_action_and_value(self):
        """get_value and get_action_and_value should return the same value for the same obs."""
        m = _model()
        obs = _rand_obs()
        with torch.no_grad():
            _, _, value_gav, _ = m.get_action_and_value(obs)
            value_gv = m.get_value(obs)
        assert torch.allclose(value_gav, value_gv, atol=1e-6)


# ---------------------------------------------------------------------------
# evaluate_actions
# ---------------------------------------------------------------------------


class TestEvaluateActions:
    def test_logp_shape(self):
        m = _model()
        obs = _rand_obs()
        with torch.no_grad():
            action, _, _, _ = m.get_action_and_value(obs)
        logp, entropy, value = m.evaluate_actions(obs, action)
        assert logp.shape == (B,)

    def test_entropy_shape(self):
        m = _model()
        obs = _rand_obs()
        with torch.no_grad():
            action, _, _, _ = m.get_action_and_value(obs)
        logp, entropy, value = m.evaluate_actions(obs, action)
        assert entropy.shape == (B,)

    def test_value_shape(self):
        m = _model()
        obs = _rand_obs()
        with torch.no_grad():
            action, _, _, _ = m.get_action_and_value(obs)
        logp, entropy, value = m.evaluate_actions(obs, action)
        assert value.shape == (B,)

    def test_entropy_nonnegative(self):
        m = _model()
        obs = _rand_obs()
        with torch.no_grad():
            action, _, _, _ = m.get_action_and_value(obs)
        _, entropy, _ = m.evaluate_actions(obs, action)
        assert (entropy >= 0).all()

    def test_logp_consistent_with_get_action_and_value(self):
        """evaluate_actions on the same (obs, action) should return the same logp."""
        m = _model()
        obs = _rand_obs()
        with torch.no_grad():
            action, logp_sample, _, _ = m.get_action_and_value(obs)
            logp_eval, _, _ = m.evaluate_actions(obs, action)
        assert torch.allclose(logp_sample, logp_eval, atol=1e-6)

    def test_entropy_bounded_above(self):
        """Entropy of Categorical(n) ≤ log(n)."""
        m = _model()
        obs = _rand_obs()
        with torch.no_grad():
            action, _, _, _ = m.get_action_and_value(obs)
        _, entropy, _ = m.evaluate_actions(obs, action)
        max_entropy = math.log(ACT_N)
        assert (entropy <= max_entropy + 1e-5).all()


# ---------------------------------------------------------------------------
# initial_state
# ---------------------------------------------------------------------------


class TestInitialState:
    def test_returns_none(self):
        m = _model()
        state = m.initial_state(batch_size=4, device=torch.device("cpu"))
        assert state is None


# ---------------------------------------------------------------------------
# Debug mode
# ---------------------------------------------------------------------------


class TestDebugMode:
    def test_debug_nan_obs_raises(self):
        m = _model(debug=True)
        obs = torch.full((B, OBS_DIM), float("nan"))
        with pytest.raises(AssertionError):
            m.get_action_and_value(obs)

    def test_debug_wrong_ndim_raises(self):
        m = _model(debug=True)
        obs = torch.randn(OBS_DIM)  # 1D instead of 2D
        with pytest.raises(AssertionError):
            m.get_action_and_value(obs)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_gain_for_tanh(self):
        import torch.nn as nn

        assert gain_for_activation("tanh") == pytest.approx(nn.init.calculate_gain("tanh"))

    def test_gain_for_relu(self):
        import torch.nn as nn

        assert gain_for_activation("relu") == pytest.approx(nn.init.calculate_gain("relu"))

    def test_gain_unknown_raises(self):
        with pytest.raises(ValueError):
            gain_for_activation("sigmoid")

    def test_activation_factory_tanh(self):
        import torch.nn as nn
        act = activation_factory("tanh")()
        assert isinstance(act, nn.Tanh)

    def test_activation_factory_relu(self):
        import torch.nn as nn
        act = activation_factory("relu")()
        assert isinstance(act, nn.ReLU)

    def test_activation_factory_unknown_raises(self):
        with pytest.raises(ValueError):
            activation_factory("sigmoid")

    def test_build_mlp_output_shape(self):
        mlp = build_mlp([4, 8, 8], activation="tanh", activate_last=True)
        x = torch.randn(3, 4)
        out = mlp(x)
        assert out.shape == (3, 8)

    def test_build_mlp_no_activate_last(self):
        """With activate_last=False, the last layer should be a Linear (no activation after it)."""
        import torch.nn as nn

        mlp = build_mlp([4, 8], activation="tanh", activate_last=False)
        assert isinstance(list(mlp.children())[-1], nn.Linear)
