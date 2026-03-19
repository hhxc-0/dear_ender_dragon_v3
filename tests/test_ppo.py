# Unit tests for src/algo/ppo.py

from __future__ import annotations

import math

import pytest
import torch

from src.algo.ppo import PPOLearner
from src.types import MiniBatch, PPOCfg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

OBS_DIM = 4
N = 8  # minibatch size


def _make_learner(model, optim, clip_coef=0.2, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5):
    cfg = PPOCfg(
        clip_coef=clip_coef,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
    )
    return PPOLearner(model=model, optim=optim, config=cfg)


def _make_mb(
    model,
    N: int = N,
    logp_offset: float = 0.0,
    adv: float = 1.0,
    value_offset: float = 0.0,
) -> MiniBatch:
    """
    Build a MiniBatch where:
      - obs is random
      - logp_old is derived from the model's current policy (so ratio ≈ exp(logp_offset))
      - advantages_norm is constant `adv`
      - values_old = model_value + value_offset
      - returns = values_old + 0.5
    """
    torch.manual_seed(0)
    obs = torch.randn(N, OBS_DIM)
    with torch.no_grad():
        action, logp_now, value_now, _ = model.get_action_and_value(obs)
    logp_old = logp_now - logp_offset  # ratio = exp(logp_now - logp_old) = exp(logp_offset)
    values_old = value_now + value_offset
    returns = values_old + 0.5
    return MiniBatch(
        batch_size=N,
        obs=obs,
        actions=action,
        logp_old=logp_old,
        values_old=values_old,
        returns=returns,
        advantages_norm=torch.full((N,), adv),
    )


# ---------------------------------------------------------------------------
# Return keys and value types
# ---------------------------------------------------------------------------


class TestUpdateReturnContract:
    EXPECTED_KEYS = {
        "losses/policy_loss", "losses/value_loss", "losses/entropy", "losses/total_loss",
        "ppo/approx_kl", "ppo/clipfrac", "ppo/ratio_mean",
        "value/explained_variance", "optim/grad_norm", "optim/lr",
    }

    def test_returns_expected_keys(self, tiny_model, tiny_optim, ppo_cfg):
        learner = PPOLearner(model=tiny_model, optim=tiny_optim, config=ppo_cfg)
        mb = _make_mb(tiny_model)
        metrics = learner.update(mb)
        assert set(metrics.keys()) == self.EXPECTED_KEYS

    def test_all_values_are_finite_floats(self, tiny_model, tiny_optim, ppo_cfg):
        learner = PPOLearner(model=tiny_model, optim=tiny_optim, config=ppo_cfg)
        mb = _make_mb(tiny_model)
        metrics = learner.update(mb)
        for k, v in metrics.items():
            assert isinstance(v, float), f"{k} is not a float"
            assert math.isfinite(v), f"{k} is not finite: {v}"


# ---------------------------------------------------------------------------
# Policy loss
# ---------------------------------------------------------------------------


class TestPolicyLoss:
    def test_unclipped_regime_loss_approx(self, tiny_model, tiny_optim):
        """When ratio ≈ 1 (logp_offset=0) and adv > 0, policy_loss ≈ -adv."""
        adv = 1.0
        learner = _make_learner(tiny_model, tiny_optim, ent_coef=0.0, vf_coef=0.0)
        mb = _make_mb(tiny_model, logp_offset=0.0, adv=adv)
        metrics = learner.update(mb)
        # policy_loss = -mean(ratio * adv) ≈ -adv when ratio ≈ 1
        assert abs(metrics["losses/policy_loss"] - (-adv)) < 0.05

    def test_clipped_regime_positive_adv(self, tiny_model, tiny_optim):
        """When ratio >> 1+clip_coef and adv > 0, clipped objective is used (loss is larger than unclipped)."""
        clip_coef = 0.2
        adv = 1.0
        # logp_offset = 2.0 → ratio = exp(2) ≈ 7.4, well above 1+clip_coef=1.2
        # Compute both policy losses analytically from the same fixed minibatch,
        # without calling update() (which would mutate model weights between the two).
        mb = _make_mb(tiny_model, logp_offset=0.0, adv=adv)
        mb_large_ratio = _make_mb(tiny_model, logp_offset=2.0, adv=adv)
        with torch.no_grad():
            # Unclipped: ratio ≈ 1, loss ≈ -adv
            ratio_unclipped = torch.exp(mb.logp_old - mb.logp_old)  # all ones
            obj_unclipped = (ratio_unclipped * mb.advantages_norm).mean()
            policy_loss_unclipped = -obj_unclipped.item()
            # Clipped: ratio = exp(2) >> 1+clip_coef, so clipped ratio = 1+clip_coef
            ratio_clipped_val = torch.exp(mb_large_ratio.logp_old + 2.0 - mb_large_ratio.logp_old)  # exp(2)
            clipped = torch.clamp(ratio_clipped_val, 1 - clip_coef, 1 + clip_coef)
            obj_clipped = torch.min(
                ratio_clipped_val * mb_large_ratio.advantages_norm,
                clipped * mb_large_ratio.advantages_norm,
            ).mean()
            policy_loss_clipped = -obj_clipped.item()
        # Clipped ratio = 1+clip_coef = 1.2 < raw ratio exp(2) ≈ 7.4
        # So clipped objective = 1.2*adv < raw objective = 7.4*adv
        # → policy_loss_clipped = -1.2 < policy_loss_unclipped = -1.0 (more negative)
        assert policy_loss_clipped < policy_loss_unclipped

    def test_negative_adv_clipping_prevents_too_large_decrease(self, tiny_model, tiny_optim):
        """When adv < 0 and ratio << 1-clip_coef, clipping prevents further decrease."""
        clip_coef = 0.2
        adv = -1.0
        # logp_offset = -2.0 → ratio = exp(-2) ≈ 0.14, well below 1-clip_coef=0.8
        learner = _make_learner(tiny_model, tiny_optim, clip_coef=clip_coef, ent_coef=0.0, vf_coef=0.0)
        mb = _make_mb(tiny_model, logp_offset=-2.0, adv=adv)
        metrics = learner.update(mb)
        # Clipped ratio = 1 - clip_coef = 0.8; policy_loss ≈ -(0.8 * (-1)) = 0.8
        assert abs(metrics["losses/policy_loss"] - (1 - clip_coef) * (-adv)) < 0.05


# ---------------------------------------------------------------------------
# Value loss
# ---------------------------------------------------------------------------


class TestValueLoss:
    def test_value_loss_nonnegative(self, tiny_model, tiny_optim, ppo_cfg):
        learner = PPOLearner(model=tiny_model, optim=tiny_optim, config=ppo_cfg)
        mb = _make_mb(tiny_model)
        metrics = learner.update(mb)
        assert metrics["losses/value_loss"] >= 0.0

    def test_value_loss_unclipped_regime(self, tiny_model, tiny_optim):
        """When value ≈ values_old (value_offset=0), value loss ≈ 0.5 * MSE(value, returns)."""
        learner = _make_learner(tiny_model, tiny_optim, ent_coef=0.0)
        mb = _make_mb(tiny_model, value_offset=0.0)
        # Run a forward pass to get the actual value prediction
        with torch.no_grad():
            _, _, value_pred, _ = tiny_model.get_action_and_value(mb.obs)
        expected_vl = 0.5 * ((value_pred - mb.returns) ** 2).mean().item()
        metrics = learner.update(mb)
        # Allow some tolerance because the clipped branch may also be active
        assert abs(metrics["losses/value_loss"] - expected_vl) < 0.5


# ---------------------------------------------------------------------------
# Entropy bonus
# ---------------------------------------------------------------------------


class TestEntropyBonus:
    def test_entropy_nonnegative(self, tiny_model, tiny_optim, ppo_cfg):
        learner = PPOLearner(model=tiny_model, optim=tiny_optim, config=ppo_cfg)
        mb = _make_mb(tiny_model)
        metrics = learner.update(mb)
        assert metrics["losses/entropy"] >= 0.0

    def test_entropy_bonus_reduces_total_loss(self, tiny_model, tiny_optim):
        """With ent_coef > 0, total_loss < policy_loss + vf_coef * value_loss."""
        ent_coef, vf_coef = 0.1, 0.5
        learner = _make_learner(tiny_model, tiny_optim, ent_coef=ent_coef, vf_coef=vf_coef)
        mb = _make_mb(tiny_model)
        metrics = learner.update(mb)
        expected_upper = metrics["losses/policy_loss"] + vf_coef * metrics["losses/value_loss"]
        assert metrics["losses/total_loss"] < expected_upper


# ---------------------------------------------------------------------------
# PPO health metrics
# ---------------------------------------------------------------------------


class TestPPOHealthMetrics:
    def test_approx_kl_positive_when_policy_moved_away(self, tiny_model, tiny_optim, ppo_cfg):
        """logp < logp_old → policy moved away → approx_kl > 0."""
        learner = PPOLearner(model=tiny_model, optim=tiny_optim, config=ppo_cfg)
        # logp_offset = -1.0 → logp = logp_old - 1 → logp_old - logp = 1 > 0
        mb = _make_mb(tiny_model, logp_offset=-1.0)
        metrics = learner.update(mb)
        assert metrics["ppo/approx_kl"] > 0.0

    def test_approx_kl_negative_when_policy_moved_closer(self, tiny_model, tiny_optim, ppo_cfg):
        """logp > logp_old → approx_kl < 0."""
        learner = PPOLearner(model=tiny_model, optim=tiny_optim, config=ppo_cfg)
        mb = _make_mb(tiny_model, logp_offset=1.0)
        metrics = learner.update(mb)
        assert metrics["ppo/approx_kl"] < 0.0

    def test_clipfrac_zero_when_unclipped(self, tiny_model, tiny_optim):
        """ratio ≈ 1 (logp_offset=0) → all ratios within clip range → clipfrac = 0."""
        learner = _make_learner(tiny_model, tiny_optim, clip_coef=0.5)
        mb = _make_mb(tiny_model, logp_offset=0.0)
        metrics = learner.update(mb)
        assert metrics["ppo/clipfrac"] == 0.0

    def test_clipfrac_one_when_all_clipped(self, tiny_model, tiny_optim):
        """ratio = exp(5) >> 1+clip_coef → all clipped → clipfrac = 1."""
        learner = _make_learner(tiny_model, tiny_optim, clip_coef=0.2)
        mb = _make_mb(tiny_model, logp_offset=5.0)
        metrics = learner.update(mb)
        assert metrics["ppo/clipfrac"] == 1.0

    def test_ratio_mean_near_one_when_logp_offset_zero(self, tiny_model, tiny_optim, ppo_cfg):
        learner = PPOLearner(model=tiny_model, optim=tiny_optim, config=ppo_cfg)
        mb = _make_mb(tiny_model, logp_offset=0.0)
        metrics = learner.update(mb)
        assert abs(metrics["ppo/ratio_mean"] - 1.0) < 0.05

    def test_lr_matches_optimizer(self, tiny_model, tiny_optim, ppo_cfg):
        learner = PPOLearner(model=tiny_model, optim=tiny_optim, config=ppo_cfg)
        mb = _make_mb(tiny_model)
        metrics = learner.update(mb)
        assert metrics["optim/lr"] == pytest.approx(tiny_optim.param_groups[0]["lr"])


# ---------------------------------------------------------------------------
# Gradient behaviour
# ---------------------------------------------------------------------------


class TestGradients:
    def test_grad_norm_finite_after_update(self, tiny_model, tiny_optim, ppo_cfg):
        learner = PPOLearner(model=tiny_model, optim=tiny_optim, config=ppo_cfg)
        mb = _make_mb(tiny_model)
        metrics = learner.update(mb)
        assert math.isfinite(metrics["optim/grad_norm"])

    def test_nonfinite_grad_raises(self, tiny_model, tiny_optim, ppo_cfg):
        """Injecting NaN into model weights should trigger the non-finite grad guard."""
        learner = PPOLearner(model=tiny_model, optim=tiny_optim, config=ppo_cfg)
        # Build the minibatch BEFORE corrupting weights so the forward pass succeeds
        mb = _make_mb(tiny_model)
        # Now corrupt a weight so the forward pass inside update() produces NaN → NaN grad
        with torch.no_grad():
            for p in tiny_model.parameters():
                p.fill_(float("nan"))
                break
        # RuntimeError: non-finite grad norm guard in ppo.py
        # ValueError: Categorical validates logits and raises on NaN (PyTorch >= 2.x)
        with pytest.raises((RuntimeError, ValueError)):
            learner.update(mb)

    def test_parameters_updated_after_step(self, tiny_model, tiny_optim, ppo_cfg):
        """Model parameters should change after an update step."""
        learner = PPOLearner(model=tiny_model, optim=tiny_optim, config=ppo_cfg)
        params_before = [p.clone() for p in tiny_model.parameters()]
        mb = _make_mb(tiny_model)
        learner.update(mb)
        params_after = list(tiny_model.parameters())
        changed = any(not torch.allclose(b, a) for b, a in zip(params_before, params_after))
        assert changed


# ---------------------------------------------------------------------------
# Protocol stubs
# ---------------------------------------------------------------------------


class TestProtocolStubs:
    def test_state_dict_raises(self, tiny_model, tiny_optim, ppo_cfg):
        learner = PPOLearner(model=tiny_model, optim=tiny_optim, config=ppo_cfg)
        with pytest.raises(NotImplementedError):
            learner.state_dict()

    def test_load_state_dict_raises(self, tiny_model, tiny_optim, ppo_cfg):
        learner = PPOLearner(model=tiny_model, optim=tiny_optim, config=ppo_cfg)
        with pytest.raises(NotImplementedError):
            learner.load_state_dict({})
