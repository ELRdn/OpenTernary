"""Numerical contract for experimental logit-based ternary STE."""

import pytest

torch = pytest.importorskip("torch")

from openternary.quant.logit_ste import hard_logit_codes, ste_logit_codes  # noqa: E402


def test_forward_hard_codes_and_gradient() -> None:
    logits = torch.tensor([-1.2, -0.55, -0.5, 0.0, 0.5, 0.55, 1.2], requires_grad=True)
    hard = hard_logit_codes(logits)
    soft = ste_logit_codes(logits)
    assert hard.tolist() == [-1, -1, 0, 0, 0, 1, 1]
    assert torch.equal(soft.detach(), hard)
    soft.sum().backward()
    assert logits.grad is not None
    assert logits.grad.tolist() == [0, 1, 1, 1, 1, 1, 0]


def test_threshold_validation() -> None:
    with pytest.raises(ValueError, match="threshold"):
        hard_logit_codes(torch.tensor([0.0]), 0.0)
