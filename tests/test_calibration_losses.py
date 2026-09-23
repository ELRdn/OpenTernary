"""Tests for calibration losses."""

import pytest


def test_streamed_backward_weights_elements_and_releases_each_graph():
    from openternary.calibration.losses import backward_reconstruction

    parameter = torch.tensor(1.0, requires_grad=True)
    first = backward_reconstruction(torch.zeros(3), parameter.expand(3), 4, "mse")
    second = backward_reconstruction(torch.zeros(1), (parameter * 2).expand(1), 4, "mse")
    assert first + second == pytest.approx(1.75)
    assert parameter.grad.item() == pytest.approx(3.5)


torch = pytest.importorskip("torch")

from openternary.calibration.losses import aggregated_mse_loss, l1_loss, mse_loss  # noqa: E402


def test_mse_basic() -> None:
    a = torch.randn(4, 4)
    b = a + 0.1
    loss = mse_loss(a, b)
    assert loss.item() > 0
    assert torch.isfinite(loss)


def test_mse_zero() -> None:
    a = torch.randn(4, 4)
    loss = mse_loss(a, a)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_mse_nan_raises() -> None:
    a = torch.tensor([1.0, float("nan")])
    b = torch.tensor([1.0, 1.0])
    with pytest.raises(ValueError, match="NaN"):
        mse_loss(a, b)


def test_l1_basic() -> None:
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([1.5, 2.5])
    loss = l1_loss(a, b)
    assert loss.item() == pytest.approx(0.5)


def test_aggregated_weighted() -> None:
    # 2 layers different sizes, weighted
    t1 = torch.randn(2, 2)  # 4 elements
    s1 = t1 + 1.0
    t2 = torch.randn(10, 10)  # 100 elements
    s2 = t2 + 0.1
    agg = aggregated_mse_loss([(t1, s1), (t2, s2)])
    # weighted should be closer to s2's error (100 elements) than simple mean
    mse1 = mse_loss(t1, s1).item()
    mse2 = mse_loss(t2, s2).item()
    simple_mean = (mse1 + mse2) / 2
    weighted = (mse1 * 4 + mse2 * 100) / 104
    assert agg.item() == pytest.approx(weighted, rel=1e-5)
    assert agg.item() != pytest.approx(simple_mean)


def test_aggregated_empty_raises() -> None:
    with pytest.raises(ValueError):
        aggregated_mse_loss([])
