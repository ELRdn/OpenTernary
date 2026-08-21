"""Seed determinism tests."""

import random

from openternary.utils.seed import seed_everything


def test_seed_everything_deterministic_stdlib() -> None:
    seed_everything(42)
    a = [random.random() for _ in range(5)]
    seed_everything(42)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_seed_everything_different_seed_produces_different_sequence() -> None:
    seed_everything(42)
    a = [random.random() for _ in range(5)]
    seed_everything(43)
    b = [random.random() for _ in range(5)]
    assert a != b


def test_seed_sets_pythonhashseed_env() -> None:
    import os

    seed_everything(123)
    assert os.environ["PYTHONHASHSEED"] == "123"


def test_seed_returns_dict() -> None:
    info = seed_everything(42)
    assert "random" in info
    assert "PYTHONHASHSEED" in info
    assert "torch" in info or "numpy" in info
