"""Soft-to-hard ternary assignment primitives."""

import pytest

torch = pytest.importorskip("torch")


def test_temperature_schedule_reaches_hard_final_region() -> None:
    from openternary.quant.soft_ternary import temperature_at_step

    assert temperature_at_step(0, 100, start=1.0, end=0.05, schedule="linear") == pytest.approx(1.0)
    assert temperature_at_step(89, 100, start=1.0, end=0.05, schedule="linear") > 0
    assert temperature_at_step(90, 100, start=1.0, end=0.05, schedule="linear") == 0.0
    assert temperature_at_step(99, 100, start=1.0, end=0.05, schedule="cosine") == 0.0


def test_nonzero_hard_fraction_reserves_a_hard_step_for_short_runs() -> None:
    from openternary.quant.soft_ternary import temperature_at_step

    assert temperature_at_step(3, 5, hard_fraction=0.1) > 0
    assert temperature_at_step(4, 5, hard_fraction=0.1) == 0.0
    assert temperature_at_step(4, 5, hard_fraction=0.0) > 0


def test_soft_assignment_converges_to_hard_ternary_codes() -> None:
    from openternary.quant.soft_ternary import harden_soft_codes, soft_ternary_codes

    weights = torch.tensor([-1.2, -0.49, 0.0, 0.51, 1.4])
    soft = soft_ternary_codes(weights, reference_scale=1.0, temperature=1e-4)
    hard = harden_soft_codes(weights, reference_scale=1.0)
    assert torch.equal(soft.round().to(torch.int8), hard)
    assert set(hard.tolist()) <= {-1, 0, 1}


def test_zero_code_logit_bias_is_minimal_optional_modulation() -> None:
    from openternary.quant.soft_ternary import soft_ternary_codes

    weights = torch.tensor([0.45, -0.45])
    unbiased = soft_ternary_codes(weights, reference_scale=1.0, temperature=0.5, zero_logit_bias=0.0)
    biased = soft_ternary_codes(weights, reference_scale=1.0, temperature=0.5, zero_logit_bias=1.0)
    assert torch.all(biased.abs() < unbiased.abs())


def test_soft_assignment_backpropagates_finite_gradient() -> None:
    from openternary.quant.soft_ternary import soft_ternary_codes

    weights = torch.tensor([-0.8, -0.2, 0.3, 0.9], requires_grad=True)
    soft = soft_ternary_codes(weights, reference_scale=1.0, temperature=0.4)
    soft.square().sum().backward()

    assert weights.grad is not None
    assert torch.isfinite(weights.grad).all()
    assert torch.count_nonzero(weights.grad).item() == weights.numel()


def test_groupwise_soft_assignment_converges_to_groupwise_hard_codes() -> None:
    from openternary.quant.soft_ternary import harden_soft_codes, soft_ternary_codes

    weights = torch.tensor([[-1.2, -0.3, 0.9, 1.1]])
    scales = torch.tensor([1.0, 2.0])
    soft = soft_ternary_codes(weights, reference_scale=scales, group_size=2, temperature=1e-4)
    hard = harden_soft_codes(weights, reference_scale=scales, group_size=2)

    assert torch.equal(soft.round().to(torch.int8), hard)
    assert hard.tolist() == [[-1, 0, 0, 1]]


def test_groupwise_zero_reference_scale_stays_exact_zero() -> None:
    from openternary.quant.soft_ternary import harden_soft_codes, soft_ternary_codes

    weights = torch.tensor([[0.0, 0.0, 0.6, -0.7]])
    scales = torch.tensor([0.0, 1.0])
    soft = soft_ternary_codes(weights, reference_scale=scales, group_size=2, temperature=0.2)
    hard = harden_soft_codes(weights, reference_scale=scales, group_size=2)

    assert torch.equal(soft[0, :2], torch.zeros(2))
    assert torch.equal(hard[0, :2], torch.zeros(2, dtype=torch.int8))


@pytest.mark.parametrize("schedule", ["linear", "cosine", "exponential"])
def test_temperature_schedules_are_positive_and_monotonic(schedule: str) -> None:
    from openternary.quant.soft_ternary import temperature_at_step

    values = [temperature_at_step(step, 100, start=1.0, end=0.05, schedule=schedule) for step in range(90)]
    assert all(value > 0 for value in values)
    assert all(left >= right for left, right in zip(values, values[1:], strict=False))
