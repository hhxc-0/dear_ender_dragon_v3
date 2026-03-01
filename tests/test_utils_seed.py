# Unit tests for src/utils/seed.py

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from src.utils.seed import seed_all


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample_torch(n=10):
    return torch.rand(n).clone()


def _sample_numpy(n=10):
    return np.random.rand(n).copy()


def _sample_python():
    return random.random()


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


class TestReproducibility:
    def test_torch_reproducible(self):
        seed_all(42)
        a = _sample_torch()
        seed_all(42)
        b = _sample_torch()
        assert torch.allclose(a, b)

    def test_numpy_reproducible(self):
        seed_all(42)
        a = _sample_numpy()
        seed_all(42)
        b = _sample_numpy()
        assert np.allclose(a, b)

    def test_python_random_reproducible(self):
        seed_all(42)
        a = _sample_python()
        seed_all(42)
        b = _sample_python()
        assert a == b

    def test_different_seeds_produce_different_torch(self):
        seed_all(42)
        a = _sample_torch()
        seed_all(123)
        b = _sample_torch()
        assert not torch.allclose(a, b)

    def test_different_seeds_produce_different_numpy(self):
        seed_all(42)
        a = _sample_numpy()
        seed_all(123)
        b = _sample_numpy()
        assert not np.allclose(a, b)

    def test_different_seeds_produce_different_python(self):
        seed_all(42)
        a = _sample_python()
        seed_all(123)
        b = _sample_python()
        assert a != b

    def test_default_seed_is_42(self):
        """seed_all() with no args should behave the same as seed_all(42)."""
        seed_all()
        a = _sample_torch()
        seed_all(42)
        b = _sample_torch()
        assert torch.allclose(a, b)


# ---------------------------------------------------------------------------
# Deterministic mode
# ---------------------------------------------------------------------------


class TestDeterministicMode:
    def test_deterministic_flag_enables_algorithms(self):
        seed_all(0, deterministic=True)
        assert torch.are_deterministic_algorithms_enabled()
        # Restore to non-deterministic for subsequent tests
        torch.use_deterministic_algorithms(False)

    def test_non_deterministic_flag_does_not_enable(self):
        # Ensure deterministic is off first
        torch.use_deterministic_algorithms(False)
        seed_all(0, deterministic=False)
        assert not torch.are_deterministic_algorithms_enabled()

    def test_deterministic_sets_cudnn_flags(self):
        seed_all(0, deterministic=True)
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
        # Restore
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
