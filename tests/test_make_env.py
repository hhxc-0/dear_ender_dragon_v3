# Unit tests for src/envs/make_env.py

from __future__ import annotations

import numpy as np
import pytest
import gymnasium as gym

from src.envs.make_env import make_env


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


ENV_ID = "CartPole-v1"


@pytest.fixture
def envs_1():
    envs = make_env(ENV_ID, n_envs=1, seed=42)
    yield envs
    envs.close()


@pytest.fixture
def envs_2():
    envs = make_env(ENV_ID, n_envs=2, seed=42)
    yield envs
    envs.close()


# ---------------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_returns_sync_vector_env(self, envs_1):
        assert isinstance(envs_1, gym.vector.SyncVectorEnv)

    def test_n_envs_1(self, envs_1):
        assert envs_1.num_envs == 1

    def test_n_envs_2(self, envs_2):
        assert envs_2.num_envs == 2

    def test_obs_space_shape_cartpole(self, envs_1):
        assert envs_1.single_observation_space.shape == (4,)

    def test_act_space_n_cartpole(self, envs_1):
        assert envs_1.single_action_space.n == 2

    def test_video_and_human_render_raises(self, tmp_path):
        with pytest.raises(AssertionError):
            make_env(ENV_ID, capture_video=True, human_render=True, video_folder=tmp_path)


# ---------------------------------------------------------------------------
# Seeding and reproducibility
# ---------------------------------------------------------------------------


class TestSeeding:
    def test_same_seed_same_initial_obs(self):
        envs_a = make_env(ENV_ID, n_envs=1, seed=7)
        envs_b = make_env(ENV_ID, n_envs=1, seed=7)
        obs_a, _ = envs_a.reset(seed=7)
        obs_b, _ = envs_b.reset(seed=7)
        envs_a.close()
        envs_b.close()
        assert np.allclose(obs_a, obs_b)

    def test_different_seeds_different_initial_obs(self):
        envs_a = make_env(ENV_ID, n_envs=1, seed=1)
        envs_b = make_env(ENV_ID, n_envs=1, seed=999)
        obs_a, _ = envs_a.reset(seed=1)
        obs_b, _ = envs_b.reset(seed=999)
        envs_a.close()
        envs_b.close()
        assert not np.allclose(obs_a, obs_b)


# ---------------------------------------------------------------------------
# Step interface
# ---------------------------------------------------------------------------


class TestStepInterface:
    def test_step_returns_five_elements(self, envs_1):
        envs_1.reset(seed=0)
        actions = envs_1.action_space.sample()
        result = envs_1.step(actions)
        assert len(result) == 5  # obs, reward, terminated, truncated, info

    def test_obs_shape_after_step(self, envs_2):
        envs_2.reset(seed=0)
        actions = envs_2.action_space.sample()
        obs, _, _, _, _ = envs_2.step(actions)
        assert obs.shape == (2, 4)

    def test_reward_shape_after_step(self, envs_2):
        envs_2.reset(seed=0)
        actions = envs_2.action_space.sample()
        _, reward, _, _, _ = envs_2.step(actions)
        assert reward.shape == (2,)

    def test_terminated_truncated_shape(self, envs_2):
        envs_2.reset(seed=0)
        actions = envs_2.action_space.sample()
        _, _, terminated, truncated, _ = envs_2.step(actions)
        assert terminated.shape == (2,)
        assert truncated.shape == (2,)


# ---------------------------------------------------------------------------
# RecordEpisodeStatistics + SAME_STEP autoreset
# ---------------------------------------------------------------------------


class TestEpisodeStatistics:
    def _run_until_done(self, envs, max_steps=1000):
        """Step until at least one episode ends; return the info dict."""
        envs.reset(seed=0)
        for _ in range(max_steps):
            actions = envs.action_space.sample()
            obs, reward, terminated, truncated, info = envs.step(actions)
            done = terminated | truncated
            if done.any():
                return obs, info, done
        pytest.skip("No episode ended within max_steps — increase max_steps or use a shorter env.")

    def test_episode_info_present_on_done(self, envs_1):
        """When an episode ends, info should contain episode statistics."""
        _, info, done = self._run_until_done(envs_1)
        # RecordEpisodeStatistics puts stats in info["final_info"] for vector envs
        assert "final_info" in info

    def test_final_obs_present_on_done(self, envs_1):
        """SAME_STEP autoreset: terminal obs is in info['final_obs'], not next_obs."""
        _, info, done = self._run_until_done(envs_1)
        # The terminal observation should be stored in final_obs
        assert "final_obs" in info or "_final_obs" in info

    def test_next_obs_is_reset_obs_on_done(self, envs_1):
        """After SAME_STEP autoreset, next_obs should be the new episode's first obs (not terminal)."""
        obs_after_done, info, done = self._run_until_done(envs_1)
        if "final_obs" in info:
            # final_obs is an object array of arrays; stack into a numeric array before indexing
            final_obs = np.stack(info["final_obs"])
            # next_obs and final_obs should differ (one is reset, one is terminal)
            assert not np.allclose(obs_after_done[done], final_obs[done])

    def test_episode_return_nonnegative_cartpole(self, envs_1):
        """CartPole gives +1 reward per step, so episode return should be ≥ 1."""
        _, info, done = self._run_until_done(envs_1)
        if "final_info" in info:
            # In Gymnasium >=1.x, final_info is a flat dict with array values per env
            fi = info["final_info"]
            if "episode" in fi:
                assert all(fi["episode"]["r"][done] >= 1.0)
