# Unit tests for src/buffers/rollout_buffer.py

from __future__ import annotations

import pytest
import torch

from src.buffers.rollout_buffer import RolloutBuffer
from src.types import MiniBatch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

T, B, OBS_DIM = 4, 2, 4


def _make_buffer(
    T: int = T,
    B: int = B,
    obs_shape: tuple[int, ...] = (OBS_DIM,),
    device: str = "cpu",
) -> RolloutBuffer:
    return RolloutBuffer(T=T, B=B, obs_shape=obs_shape, device=device)


def _step_kwargs(**overrides):
    """Return keyword arguments for a single buf.add() call."""
    base = dict(
        obs=torch.randn(B, OBS_DIM),
        action=torch.zeros(B, dtype=torch.int64),
        logp=torch.full((B,), -0.693),
        value=torch.ones(B) * 0.5,
        next_value=torch.zeros(B),
        reward=torch.ones(B),
        terminated=torch.zeros(B, dtype=torch.bool),
        truncated=torch.zeros(B, dtype=torch.bool),
        done=torch.zeros(B, dtype=torch.bool),
        timeout=torch.zeros(B, dtype=torch.bool),
        episode_start=torch.zeros(B, dtype=torch.bool),
    )
    base.update(overrides)
    return base


def _fill_buffer(buf: RolloutBuffer, **step_overrides) -> None:
    """Fill buf to capacity with identical steps."""
    for _ in range(buf.T):
        buf.add(**_step_kwargs(**step_overrides))


# ---------------------------------------------------------------------------
# reset / add
# ---------------------------------------------------------------------------


class TestResetAndAdd:
    def test_reset_clears_pointer(self):
        buf = _make_buffer()
        _fill_buffer(buf)
        buf.reset()
        assert buf.t == 0

    def test_reset_clears_derived_fields(self):
        buf = _make_buffer()
        _fill_buffer(buf)
        buf.finalize(torch.zeros(B))
        buf.compute_returns_and_advantages(0.99, 0.95, False, 1e-8)
        buf.reset()
        assert buf.advantages is None
        assert buf.returns is None
        assert buf.advantages_norm is None

    def test_add_increments_pointer(self):
        buf = _make_buffer()
        buf.add(**_step_kwargs())
        assert buf.t == 1

    def test_add_T_times_fills_buffer(self):
        buf = _make_buffer()
        _fill_buffer(buf)
        assert buf.t == T

    def test_add_overflow_raises(self):
        buf = _make_buffer()
        _fill_buffer(buf)
        with pytest.raises(AssertionError):
            buf.add(**_step_kwargs())

    def test_add_stores_obs(self):
        buf = _make_buffer()
        obs = torch.arange(B * OBS_DIM, dtype=torch.float32).reshape(B, OBS_DIM)
        buf.add(**_step_kwargs(obs=obs))
        assert torch.allclose(buf.obs[0], obs)

    def test_add_stores_reward(self):
        buf = _make_buffer()
        reward = torch.tensor([3.0, 7.0])
        buf.add(**_step_kwargs(reward=reward))
        assert torch.allclose(buf.rewards[0], reward)


# ---------------------------------------------------------------------------
# finalize
# ---------------------------------------------------------------------------


class TestFinalize:
    def test_finalize_fills_last_step_from_last_value(self):
        buf = _make_buffer()
        values = [torch.full((B,), float(t)) for t in range(T)]
        for t in range(T):
            buf.add(**_step_kwargs(value=values[t]))
        last_value = torch.full((B,), 99.0)
        buf.finalize(last_value)
        # Last step is non-done → next_values[T-1] should come from last_value
        assert torch.allclose(buf.next_values[T - 1], last_value)

    def test_finalize_fills_intermediate_from_next_values(self):
        buf = _make_buffer()
        values = [torch.full((B,), float(t)) for t in range(T)]
        for t in range(T):
            buf.add(**_step_kwargs(value=values[t]))
        last_value = torch.zeros(B)
        buf.finalize(last_value)
        # Non-done intermediate steps: next_values[t] == values[t+1]
        for t in range(T - 1):
            assert torch.allclose(buf.next_values[t], values[t + 1])

    def test_finalize_preserves_done_next_values(self):
        """For a done step, next_value was set during add() and must not be overwritten."""
        buf = _make_buffer()
        done_next_value = torch.full((B,), 42.0)
        # Step 0: done, next_value explicitly set
        buf.add(**_step_kwargs(
            done=torch.ones(B, dtype=torch.bool),
            terminated=torch.ones(B, dtype=torch.bool),
            next_value=done_next_value,
        ))
        # Remaining steps: non-done
        for _ in range(T - 1):
            buf.add(**_step_kwargs())
        buf.finalize(torch.zeros(B))
        assert torch.allclose(buf.next_values[0], done_next_value)

    def test_finalize_no_nan_after_all_non_done(self):
        buf = _make_buffer()
        _fill_buffer(buf)
        buf.finalize(torch.ones(B))
        assert torch.isfinite(buf.next_values[:T]).all()


# ---------------------------------------------------------------------------
# compute_returns_and_advantages
# ---------------------------------------------------------------------------


class TestGAE:
    def _single_step_buf(self, reward, value, next_value, terminated, truncated):
        """Helper: 1-step buffer with a single env."""
        buf = RolloutBuffer(T=1, B=1, obs_shape=(1,), device="cpu")
        done = terminated | truncated
        timeout = truncated & ~terminated
        buf.add(
            obs=torch.zeros(1, 1),
            action=torch.zeros(1, dtype=torch.int64),
            logp=torch.zeros(1),
            value=torch.tensor([value]),
            next_value=torch.tensor([next_value]),
            reward=torch.tensor([reward]),
            terminated=terminated.unsqueeze(0),
            truncated=truncated.unsqueeze(0),
            done=done.unsqueeze(0),
            timeout=timeout.unsqueeze(0),
            episode_start=torch.zeros(1, dtype=torch.bool),
        )
        # finalize: done step → next_value already set; last_value irrelevant
        buf.finalize(torch.tensor([next_value]))
        buf.compute_returns_and_advantages(gamma=0.99, gae_lambda=1.0, normalize_advantages=False, eps=1e-8)
        return buf

    def test_terminal_no_bootstrap(self):
        """Terminated step: TD target = reward only (no gamma * next_value)."""
        reward, value, next_value = 1.0, 0.5, 10.0
        terminated = torch.tensor(True)
        truncated = torch.tensor(False)
        buf = self._single_step_buf(reward, value, next_value, terminated, truncated)
        expected_adv = reward - value  # no bootstrap
        assert buf.advantages is not None
        assert torch.isclose(buf.advantages[0, 0], torch.tensor(expected_adv), atol=1e-5)

    def test_timeout_bootstraps(self):
        """Timeout truncation: TD target includes gamma * next_value."""
        reward, value, next_value = 1.0, 0.5, 2.0
        terminated = torch.tensor(False)
        truncated = torch.tensor(True)
        buf = self._single_step_buf(reward, value, next_value, terminated, truncated)
        expected_adv = reward + 0.99 * next_value - value
        assert buf.advantages is not None
        assert torch.isclose(buf.advantages[0, 0], torch.tensor(expected_adv), atol=1e-5)

    def test_non_done_bootstraps(self):
        """Non-done step: TD target includes gamma * next_value."""
        reward, value, next_value = 1.0, 0.5, 2.0
        terminated = torch.tensor(False)
        truncated = torch.tensor(False)
        buf = self._single_step_buf(reward, value, next_value, terminated, truncated)
        expected_adv = reward + 0.99 * next_value - value
        assert buf.advantages is not None
        assert torch.isclose(buf.advantages[0, 0], torch.tensor(expected_adv), atol=1e-5)

    def test_gae_trace_propagates_across_non_done_steps(self):
        """Two non-done steps: adv[0] should include discounted adv[1]."""
        gamma, lam = 0.99, 0.95
        buf = RolloutBuffer(T=2, B=1, obs_shape=(1,), device="cpu")
        reward = 1.0
        value = 0.5
        next_value_step1 = 2.0  # used for step 1 bootstrap

        for _ in range(2):
            buf.add(
                obs=torch.zeros(1, 1),
                action=torch.zeros(1, dtype=torch.int64),
                logp=torch.zeros(1),
                value=torch.tensor([value]),
                next_value=torch.zeros(1),
                reward=torch.tensor([reward]),
                terminated=torch.zeros(1, dtype=torch.bool),
                truncated=torch.zeros(1, dtype=torch.bool),
                done=torch.zeros(1, dtype=torch.bool),
                timeout=torch.zeros(1, dtype=torch.bool),
                episode_start=torch.zeros(1, dtype=torch.bool),
            )
        buf.finalize(torch.tensor([next_value_step1]))
        buf.compute_returns_and_advantages(gamma=gamma, gae_lambda=lam, normalize_advantages=False, eps=1e-8)

        # Manual GAE:
        # delta[1] = reward + gamma * next_value_step1 - value
        delta1 = reward + gamma * next_value_step1 - value
        adv1 = delta1  # last step
        # delta[0] = reward + gamma * value - value  (next_value[0] = values[1] = value)
        delta0 = reward + gamma * value - value
        adv0 = delta0 + gamma * lam * adv1

        assert buf.advantages is not None
        assert torch.isclose(buf.advantages[1, 0], torch.tensor(adv1), atol=1e-5)
        assert torch.isclose(buf.advantages[0, 0], torch.tensor(adv0), atol=1e-5)

    def test_done_cuts_gae_trace(self):
        """A done at step 0 must prevent adv[0] from including any contribution from step 1."""
        gamma, lam = 0.99, 0.95
        buf = RolloutBuffer(T=2, B=1, obs_shape=(1,), device="cpu")
        # Step 0: terminal done
        buf.add(
            obs=torch.zeros(1, 1),
            action=torch.zeros(1, dtype=torch.int64),
            logp=torch.zeros(1),
            value=torch.tensor([0.5]),
            next_value=torch.zeros(1),  # terminal → no bootstrap
            reward=torch.tensor([1.0]),
            terminated=torch.ones(1, dtype=torch.bool),
            truncated=torch.zeros(1, dtype=torch.bool),
            done=torch.ones(1, dtype=torch.bool),
            timeout=torch.zeros(1, dtype=torch.bool),
            episode_start=torch.zeros(1, dtype=torch.bool),
        )
        # Step 1: non-done with large reward to make contamination obvious
        buf.add(
            obs=torch.zeros(1, 1),
            action=torch.zeros(1, dtype=torch.int64),
            logp=torch.zeros(1),
            value=torch.tensor([0.5]),
            next_value=torch.zeros(1),
            reward=torch.tensor([1000.0]),
            terminated=torch.zeros(1, dtype=torch.bool),
            truncated=torch.zeros(1, dtype=torch.bool),
            done=torch.zeros(1, dtype=torch.bool),
            timeout=torch.zeros(1, dtype=torch.bool),
            episode_start=torch.zeros(1, dtype=torch.bool),
        )
        buf.finalize(torch.tensor([0.5]))
        buf.compute_returns_and_advantages(gamma=gamma, gae_lambda=lam, normalize_advantages=False, eps=1e-8)

        # adv[0] = reward[0] - value[0] (terminal, no bootstrap, no GAE carry)
        expected_adv0 = 1.0 - 0.5
        assert buf.advantages is not None
        assert torch.isclose(buf.advantages[0, 0], torch.tensor(expected_adv0), atol=1e-5)

    def test_timeout_mid_rollout_uses_terminal_value_not_reset_value(self):
        """Regression: a timeout at step 0 of a 3-step rollout must bootstrap
        with V(terminal_obs), not V(reset_obs) stored in values[1]."""
        gamma, lam = 0.99, 1.0
        buf = RolloutBuffer(T=3, B=1, obs_shape=(1,), device="cpu")

        # Step 0: timeout truncation
        #   value=0.5 (current state), next_value=5.0 (V(terminal_obs) before reset)
        buf.add(
            obs=torch.zeros(1, 1),
            action=torch.zeros(1, dtype=torch.int64),
            logp=torch.zeros(1),
            value=torch.tensor([0.5]),
            next_value=torch.tensor([5.0]),
            reward=torch.tensor([1.0]),
            terminated=torch.zeros(1, dtype=torch.bool),
            truncated=torch.ones(1, dtype=torch.bool),
            done=torch.ones(1, dtype=torch.bool),
            timeout=torch.ones(1, dtype=torch.bool),
            episode_start=torch.zeros(1, dtype=torch.bool),
        )
        # Step 1: new episode after auto-reset (value=0.1, very different from 5.0)
        buf.add(
            obs=torch.zeros(1, 1),
            action=torch.zeros(1, dtype=torch.int64),
            logp=torch.zeros(1),
            value=torch.tensor([0.1]),
            next_value=torch.zeros(1),
            reward=torch.tensor([1.0]),
            terminated=torch.zeros(1, dtype=torch.bool),
            truncated=torch.zeros(1, dtype=torch.bool),
            done=torch.zeros(1, dtype=torch.bool),
            timeout=torch.zeros(1, dtype=torch.bool),
            episode_start=torch.ones(1, dtype=torch.bool),
        )
        # Step 2: continuation
        buf.add(
            obs=torch.zeros(1, 1),
            action=torch.zeros(1, dtype=torch.int64),
            logp=torch.zeros(1),
            value=torch.tensor([0.2]),
            next_value=torch.zeros(1),
            reward=torch.tensor([1.0]),
            terminated=torch.zeros(1, dtype=torch.bool),
            truncated=torch.zeros(1, dtype=torch.bool),
            done=torch.zeros(1, dtype=torch.bool),
            timeout=torch.zeros(1, dtype=torch.bool),
            episode_start=torch.zeros(1, dtype=torch.bool),
        )
        buf.finalize(torch.tensor([0.3]))
        buf.compute_returns_and_advantages(
            gamma=gamma, gae_lambda=lam, normalize_advantages=False, eps=1e-8
        )

        # delta_0 = r + gamma * V(terminal) * bootstrap - V(s_0)
        #         = 1.0 + 0.99 * 5.0 * 1.0 - 0.5 = 5.45
        # continue_gae[0] = 0.0 (done), so gae_0 = delta_0 = 5.45
        expected_adv0 = 1.0 + gamma * 5.0 - 0.5
        assert buf.advantages is not None
        assert torch.isclose(
            buf.advantages[0, 0], torch.tensor(expected_adv0), atol=1e-5
        ), (
            f"Expected {expected_adv0}, got {buf.advantages[0, 0].item()}. "
            f"If ~0.599, the bug is using V(reset)=0.1 instead of V(terminal)=5.0"
        )

    def test_timeout_mid_rollout_multiple_envs(self):
        """Timeout in env 0 at step 1, no done in env 1 — each env independent."""
        gamma, lam = 0.99, 1.0
        buf = RolloutBuffer(T=3, B=2, obs_shape=(1,), device="cpu")

        def add(value, next_value, reward, done, timeout, episode_start):
            buf.add(
                obs=torch.zeros(2, 1),
                action=torch.zeros(2, dtype=torch.int64),
                logp=torch.zeros(2),
                value=value,
                next_value=next_value,
                reward=reward,
                terminated=torch.zeros(2, dtype=torch.bool),
                truncated=done,
                done=done,
                timeout=timeout,
                episode_start=episode_start,
            )

        # Step 0: both envs normal
        add(
            value=torch.tensor([1.0, 1.0]),
            next_value=torch.zeros(2),
            reward=torch.tensor([1.0, 1.0]),
            done=torch.zeros(2, dtype=torch.bool),
            timeout=torch.zeros(2, dtype=torch.bool),
            episode_start=torch.zeros(2, dtype=torch.bool),
        )
        # Step 1: env 0 timeouts with V(terminal)=8.0, env 1 continues
        add(
            value=torch.tensor([2.0, 2.0]),
            next_value=torch.tensor([8.0, 0.0]),
            reward=torch.tensor([1.0, 1.0]),
            done=torch.tensor([True, False]),
            timeout=torch.tensor([True, False]),
            episode_start=torch.zeros(2, dtype=torch.bool),
        )
        # Step 2: env 0 new episode (V(reset)=0.1), env 1 continues
        add(
            value=torch.tensor([0.1, 3.0]),
            next_value=torch.zeros(2),
            reward=torch.tensor([1.0, 1.0]),
            done=torch.zeros(2, dtype=torch.bool),
            timeout=torch.zeros(2, dtype=torch.bool),
            episode_start=torch.tensor([True, False]),
        )
        buf.finalize(torch.tensor([0.5, 0.5]))
        buf.compute_returns_and_advantages(
            gamma=gamma, gae_lambda=lam, normalize_advantages=False, eps=1e-8
        )

        # Env 0, step 1: delta = 1.0 + 0.99 * 8.0 - 2.0 = 6.92
        # continue_gae=0 (done), so adv = 6.92
        expected = 1.0 + gamma * 8.0 - 2.0
        assert buf.advantages is not None
        assert torch.isclose(buf.advantages[1, 0], torch.tensor(expected), atol=1e-5), (
            f"Env 0 step 1: expected {expected}, got {buf.advantages[1, 0].item()}"
        )

        # Env 1, step 1: no done, next_values[1] = values[2] = 3.0 (from finalize)
        # delta = 1.0 + 0.99 * 3.0 - 2.0 = 1.97
        # gae propagates from step 2
        delta2_env1 = 1.0 + gamma * 0.5 - 3.0  # step 2 bootstrap from last_value
        delta1_env1 = 1.0 + gamma * 3.0 - 2.0
        expected_env1 = delta1_env1 + gamma * lam * delta2_env1
        assert torch.isclose(buf.advantages[1, 1], torch.tensor(expected_env1), atol=1e-5), (
            f"Env 1 step 1: expected {expected_env1}, got {buf.advantages[1, 1].item()}"
        )

    def test_consecutive_timeouts_both_use_correct_terminal_values(self):
        """Two consecutive timeouts at steps 0 and 1 — each must use its own V(terminal)."""
        gamma, lam = 0.99, 1.0
        buf = RolloutBuffer(T=3, B=1, obs_shape=(1,), device="cpu")

        # Step 0: timeout, V(terminal)=4.0
        buf.add(
            obs=torch.zeros(1, 1),
            action=torch.zeros(1, dtype=torch.int64),
            logp=torch.zeros(1),
            value=torch.tensor([1.0]),
            next_value=torch.tensor([4.0]),
            reward=torch.tensor([1.0]),
            terminated=torch.zeros(1, dtype=torch.bool),
            truncated=torch.ones(1, dtype=torch.bool),
            done=torch.ones(1, dtype=torch.bool),
            timeout=torch.ones(1, dtype=torch.bool),
            episode_start=torch.zeros(1, dtype=torch.bool),
        )
        # Step 1: also timeout (new short episode), V(terminal)=7.0
        buf.add(
            obs=torch.zeros(1, 1),
            action=torch.zeros(1, dtype=torch.int64),
            logp=torch.zeros(1),
            value=torch.tensor([0.2]),
            next_value=torch.tensor([7.0]),
            reward=torch.tensor([2.0]),
            terminated=torch.zeros(1, dtype=torch.bool),
            truncated=torch.ones(1, dtype=torch.bool),
            done=torch.ones(1, dtype=torch.bool),
            timeout=torch.ones(1, dtype=torch.bool),
            episode_start=torch.ones(1, dtype=torch.bool),
        )
        # Step 2: new episode after second reset
        buf.add(
            obs=torch.zeros(1, 1),
            action=torch.zeros(1, dtype=torch.int64),
            logp=torch.zeros(1),
            value=torch.tensor([0.3]),
            next_value=torch.zeros(1),
            reward=torch.tensor([1.0]),
            terminated=torch.zeros(1, dtype=torch.bool),
            truncated=torch.zeros(1, dtype=torch.bool),
            done=torch.zeros(1, dtype=torch.bool),
            timeout=torch.zeros(1, dtype=torch.bool),
            episode_start=torch.ones(1, dtype=torch.bool),
        )
        buf.finalize(torch.tensor([0.5]))
        buf.compute_returns_and_advantages(
            gamma=gamma, gae_lambda=lam, normalize_advantages=False, eps=1e-8
        )

        assert buf.advantages is not None
        # Step 1: delta = 2.0 + 0.99*7.0 - 0.2 = 8.73, gae cut by done
        expected1 = 2.0 + gamma * 7.0 - 0.2
        assert torch.isclose(buf.advantages[1, 0], torch.tensor(expected1), atol=1e-5), (
            f"Step 1: expected {expected1}, got {buf.advantages[1, 0].item()}"
        )
        # Step 0: delta = 1.0 + 0.99*4.0 - 1.0 = 3.96, gae cut by done
        expected0 = 1.0 + gamma * 4.0 - 1.0
        assert torch.isclose(buf.advantages[0, 0], torch.tensor(expected0), atol=1e-5), (
            f"Step 0: expected {expected0}, got {buf.advantages[0, 0].item()}"
        )

    def test_returns_equal_advantages_plus_values(self, filled_buffer):
        assert torch.allclose(
            filled_buffer.returns,
            filled_buffer.advantages + filled_buffer.values[:T],
            atol=1e-6,
        )

    def test_advantage_normalization_mean_zero(self):
        from tests.conftest import _make_filled_buffer
        buf = _make_filled_buffer(normalize_advantages=True)
        assert buf.advantages_norm is not None
        adv_flat = buf.advantages_norm.reshape(-1)
        assert torch.isclose(adv_flat.mean(), torch.tensor(0.0), atol=1e-5)

    def test_advantage_normalization_std_one(self):
        from tests.conftest import _make_filled_buffer
        buf = _make_filled_buffer(normalize_advantages=True)
        assert buf.advantages_norm is not None
        adv_flat = buf.advantages_norm.reshape(-1)
        assert torch.isclose(adv_flat.std(unbiased=False), torch.tensor(1.0), atol=1e-5)

    def test_advantage_normalization_constant_no_nan(self):
        """All-zero advantages (zero variance) must not produce NaN after normalization."""
        buf = RolloutBuffer(T=2, B=2, obs_shape=(1,), device="cpu")
        for _ in range(2):
            buf.add(
                obs=torch.zeros(2, 1),
                action=torch.zeros(2, dtype=torch.int64),
                logp=torch.zeros(2),
                value=torch.ones(2),
                next_value=torch.zeros(2),
                reward=torch.ones(2),  # constant reward → constant advantages
                terminated=torch.zeros(2, dtype=torch.bool),
                truncated=torch.zeros(2, dtype=torch.bool),
                done=torch.zeros(2, dtype=torch.bool),
                timeout=torch.zeros(2, dtype=torch.bool),
                episode_start=torch.zeros(2, dtype=torch.bool),
            )
        buf.finalize(torch.ones(2))
        buf.compute_returns_and_advantages(gamma=0.0, gae_lambda=0.0, normalize_advantages=True, eps=1e-8)
        assert buf.advantages_norm is not None
        assert torch.isfinite(buf.advantages_norm).all()


# ---------------------------------------------------------------------------
# iter_minibatches
# ---------------------------------------------------------------------------


class TestIterMinibatches:
    def test_raises_before_compute(self):
        buf = _make_buffer()
        _fill_buffer(buf)
        buf.finalize(torch.zeros(B))
        with pytest.raises(RuntimeError):
            list(buf.iter_minibatches(mini_batch_size=4, shuffle=False, device="cpu"))

    def test_covers_all_data(self, filled_buffer):
        total = sum(mb.batch_size for mb in filled_buffer.iter_minibatches(4, False, "cpu"))
        assert total == T * B

    def test_minibatch_obs_shape(self, filled_buffer):
        for mb in filled_buffer.iter_minibatches(4, False, "cpu"):
            assert mb.obs.shape == (mb.batch_size, OBS_DIM)

    def test_minibatch_actions_shape(self, filled_buffer):
        for mb in filled_buffer.iter_minibatches(4, False, "cpu"):
            assert mb.actions.shape == (mb.batch_size,)

    def test_minibatch_scalar_fields_shape(self, filled_buffer):
        for mb in filled_buffer.iter_minibatches(4, False, "cpu"):
            for field in (mb.logp_old, mb.values_old, mb.returns, mb.advantages_norm):
                assert field.shape == (mb.batch_size,)

    def test_no_shuffle_deterministic(self, filled_buffer):
        obs_a = torch.cat([mb.obs for mb in filled_buffer.iter_minibatches(4, False, "cpu")])
        obs_b = torch.cat([mb.obs for mb in filled_buffer.iter_minibatches(4, False, "cpu")])
        assert torch.allclose(obs_a, obs_b)

    def test_shuffle_randomizes_order(self, filled_buffer):
        torch.manual_seed(1)
        obs_a = torch.cat([mb.obs for mb in filled_buffer.iter_minibatches(2, True, "cpu")])
        torch.manual_seed(99)
        obs_b = torch.cat([mb.obs for mb in filled_buffer.iter_minibatches(2, True, "cpu")])
        assert not torch.allclose(obs_a, obs_b)

    def test_minibatch_is_named_tuple(self, filled_buffer):
        for mb in filled_buffer.iter_minibatches(4, False, "cpu"):
            assert isinstance(mb, MiniBatch)
